import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import List from '@material-ui/core/List';
import Box from '@material-ui/core/Box';
import ListSubheader from '@material-ui/core/ListSubheader';
import StockListItem from './StockListItem'
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';
import { FixedSizeList } from 'react-window';

const useStyles = makeStyles((theme) => ({
  stockComponent: {
    margin: theme.spacing(0, 0, 2),
  },
  listSubHeader: {
    textAlign: 'initial'
  },
}));


function StockSelectedList(props) {
  const { selectedStocks } = props
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
          >
          </StockListItem>
        }
      </div>
    );
  };

  return (
    <Box className={classes.stockComponent}>
      <FixedSizeList
        height={350}
        itemSize={60}
        itemCount={tempSelectedStocks.length}
        itemData={tempSelectedStocks}
      >
        {renderRow}
      </FixedSizeList >
    </Box>
  );
}

export default StockSelectedList