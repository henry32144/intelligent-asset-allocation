import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import List from '@material-ui/core/List';
import Box from '@material-ui/core/Box';
import ListSubheader from '@material-ui/core/ListSubheader';
import StockListItem from './StockListItem'
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';
import { FixedSizeList } from 'react-window';
import Typography from '@material-ui/core/Typography';
import Divider from '@material-ui/core/Divider';

const useStyles = makeStyles((theme) => ({
  stockComponent: {
    margin: theme.spacing(0, 0, 2),
  },
  listSubHeader: {
    textAlign: 'initial'
  },
  listTitle: {
    margin: theme.spacing(1, 1, 1),
  },
  emptyText: {
    margin: theme.spacing(2, 0, 2),
    textAlign: 'center'
  }
}));


function StockSelectedList(props) {
  const classes = useStyles();

  const tempSelectedStocks = [{
    'companyName':'Apple',
    'companySymbol': 'Apple',
    'companyId':'Apple',
  },
  {
    'companyName':'BANANA',
    'companySymbol': 'BANANA',
    'companyId':'BANANA',
  }
  ];

  const removeSelectedStock = (id) => {
    var selectedStocks = Array.from(props.selectedStocks);
    var index = selectedStocks.findIndex(x => x.companyId === id);
    if (index !== -1) {
      selectedStocks.splice(index, 1);
      props.setSelectedStocks(selectedStocks);
    }
  };

  const stockListItemOnclick = (symbol) => {
    console.log(symbol);
    if (symbol === props.currentSelectedStock) {
      props.setCurrentSelectedStock(symbol);
    }
  }

  const renderRow = (props) => {
    const { data, index, style } = props;
    const rowItem = data[index];
    return (
      <div style={style}>
        {
          <StockListItem
            companyName={rowItem.companyName}
            companySymbol={rowItem.companySymbol}
            companyId={rowItem.companyId}
            removeSelectedStock={removeSelectedStock}
            stockListItemOnclick={stockListItemOnclick}
          >
          </StockListItem>
        }
      </div>
    );
  };

  return (
    <Box className={classes.stockComponent}>
      <Typography variant="h6" className={classes.listTitle}>
        Selected Stocks
      </Typography>
      <Divider></Divider>
      { props.selectedStocks.length > 0 ?
      <FixedSizeList
        height={350}
        itemSize={60}
        itemCount={props.selectedStocks.length}
        itemData={props.selectedStocks}
      >
        {renderRow}
      </FixedSizeList >
      :
      <Typography className={classes.emptyText}>
        This portfolio is empty
      </Typography>
      }
    </Box>
  );
}

export default StockSelectedList