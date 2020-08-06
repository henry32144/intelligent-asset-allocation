import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Autocomplete from '@material-ui/lab/Autocomplete';
import TextField from '@material-ui/core/TextField';
import matchSorter from 'match-sorter'

const useStyles = makeStyles((theme) => ({
  searchBox: {
    display: 'flex',
    alignItems: 'center',
    minWidth: 300,
  },
}));

function StockSearchBox(props) {
  const [value, setValue] = React.useState(null);

  const { additionalStyles, companyData, selectedStocks, setSelectedStocks } = props
  const classes = useStyles();
  const filterOptions = (options, { inputValue }) => {
    var filted = matchSorter(options, inputValue, { keys: ['companyName', 'companySymbol'] });
    return filted.sort((a, b) => {
      if (a.volatility === b.volatility) {
        return 0;
      }
      if (a.volatility === "stable") {
        return -1
      }
      if (b.volatility === "stable") {
        return 1
      }
    });
  };

  const stockOnSelected = (event, newValue) => {
    addStockToPortfolio(newValue);
  };

  const addStockToPortfolio = (newValue) => {
    console.log(newValue)
    if (selectedStocks.find(x => x.companyId === newValue.companyId) != null) {
      props.setDialogTitle("Error")
      props.setDialogMessage("The stock is already in the list");
      props.openMessageDialog();
    } else {
      var newPortfolioStocks = selectedStocks.map(function(item, index, array){
        return item.companySymbol;
      });
      setSelectedStocks([...newPortfolioStocks, newValue.companySymbol]);
      props.setDialogTitle("Success")
      props.setDialogMessage("Add " + newValue.companyName + " to your portfolio");
      props.openMessageDialog();
    }
  };

  return (
    <div>
      <Autocomplete
        freeSolo
        className={classes.searchBox}
        id="search-box"
        disableClearable
        size="small"
        groupBy={(option) => option.volatility}
        onChange={stockOnSelected}
        options={companyData.sort((a, b) => {
          if (a.volatility === b.volatility) {
            return 0;
          }
          if (a.volatility === "stable") {
            return -1
          }
          if (b.volatility === "stable") {
            return 1
          }
        })}
        getOptionLabel={(option) => option.companyName}
        renderOption={(option) => (
          <React.Fragment>
            {option.companyName} ({option.companySymbol})
          </React.Fragment>
        )}
        filterOptions={filterOptions}
        renderInput={(params) => (
          <TextField
            {...params}
            label="Add company"
            margin="normal"
            variant="outlined"
            InputProps={{ ...params.InputProps, type: 'search' }}
          />
        )}
      />
    </div>
  );
}

export default StockSearchBox;